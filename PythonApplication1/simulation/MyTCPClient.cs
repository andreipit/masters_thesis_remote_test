using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.IO;

public class MyTCPClient : MonoBehaviour
{
    void Start()
    {
		Debug.Log("start");
    }

	void Update()
	{
		if (Input.GetKeyDown(KeyCode.Space))
        {
            byte[] mes = System.Text.Encoding.ASCII.GetBytes("{key1:val1, key2:val2}"); // string someString = Encoding.ASCII.GetString(bytes);
			MyTCPClientLibrary.Send(mes);
        }
	}
}

public static class MyTCPClientLibrary
{

	#region Public static methods

	public static void Send(byte[] _Mes)
	{
		var request = CreateRequest();
		SendRequest(request, _Mes);
		string result = GetResponse(request);
		Debug.Log(result);
	}

	#endregion


	#region Private static methods

	static WebRequest CreateRequest()
    {
		string url = @"http://127.0.0.1:4567";
		var res = WebRequest.Create(url);
		res.PreAuthenticate = true;
		res.Method = "PUT";
		res.ContentType = "application/json";
		res.Headers.Add("Authorization", "Basic ");
		return res;
	}

	static void SendRequest(WebRequest _Request, byte[] _Mes)
    {
		Stream dataStream = _Request.GetRequestStream();
		dataStream.Write(_Mes, 0, _Mes.Length);
		dataStream.Close();
	}

	static string GetResponse(WebRequest _Request)
    {
		WebResponse response = _Request.GetResponse();
		StreamReader sr = new StreamReader(response.GetResponseStream());
		string result = "";
		while (sr.Peek() != -1)
			result += sr.ReadToEnd();
		//Debug.Log("header= " + response.Headers);
		response.Close();
		return result;
	}

	#endregion

}
